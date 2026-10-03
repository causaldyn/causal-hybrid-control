# ADR 0049 — A prescription evaluated as a target trial

**Status:** proposed, 2026-10-03.

## Context

`Prescription.evaluate(panel)` (ADR 0023) estimates what the prescribed schedule would cost over its
horizon, from a panel the plan did not act on, by per-decision importance sampling over windows of
`H + 1` consecutive periods. Read as the emulation of a target trial (Hernán and Robins, *Causal
Inference: What If*, chapter 22), each window is a unit enrolled at the window's first period,
assigned the plan or the policy in place, and followed for `H` periods. Three parts of that trial
were wrong.

- **Time zero.** Each unit's windows were cut back from its own latest period, so where a unit's
  record ends decided where each of its windows starts. Section 22.4 puts time zero where
  eligibility is met and follow-up starts: "study outcomes begin to be counted after that point but
  not earlier." A unit that leaves the panel because of what happened to it has its windows end
  where it left, and the windows before are chosen by their own outcomes. A zone closed three
  periods after its supply first strayed gives, cut back, the windows in which it had not yet
  strayed, and the estimate read off them is below the plan's value from their starts.
- **The interval.** A unit's windows follow each other and share its state, and the delta method
  over windows took them as independent; ADR 0023 says so among its consequences. Section 22.4,
  on emulating a trial at each of several times: "because individuals may be included in multiple
  target trials, appropriate adjustment of the variance of the effect estimate is required. This
  can be achieved by bootstrapping the entire analysis."
- **The comparator.** The evaluation reported the plan's value alone. A trial compares strategies,
  and the decision to deploy compares the plan with the policy in place, which can do better than
  the plan, and better than any fixed alternative a trial would set against it (Fine Points 22.7
  and 22.8). The plan's value with nothing beside it does not say whether switching helps, and two
  marginal intervals read side by side count twice what the two values share.

## Decision

- **Time zeros on the calendar.** `Prescription.evaluate(panel, *, time_zero="calendar")`: the
  time zeros are the panel's periods `H`, `2H`, ... before its last, the same for every unit, and
  a unit gives the window starting at each time zero when it is observed at all `H + 1` of the
  window's periods. Consecutive windows share their boundary period, as before, and the most recent
  periods are read for every unit observed to the end.
  - What decides whether a unit is in a window is its being observed to the window's end. When a
    unit's leaving is settled `H` or more periods ahead, that depends on its history before the
    time zero alone, which moves the law the windows start from and nothing after it; the value is
    read at that law, as before.
  - `time_zero="unit"` keeps the cut-back windows, for comparison.
- **A bootstrap over the units, the whole analysis inside.** The intervals come from
  `n_resamples=200` draws of the units with replacement, from `seed`, every window of a drawn unit
  kept. Each draw runs again everything that reads the windows: the plant's linearisation and its
  noise, the logger's fit unless one is given, the starts' law, the smoothing unless one is given,
  the certificate, and both values.
  - An interval is the estimate on every unit, plus or minus 1.96 standard deviations of its
    draws, widened by `model_error` times the model's correction as `evaluate_plan` widens its own.
  - A draw the certificate refuses is left out, counted (`PlanEvaluation.bootstrap`, a
    `UnitBootstrap`) and logged at `WARNING` (`chc_event="unit_bootstrap"`). When every draw but
    one is refused, no spread is left and the evaluation raises `InfeasibleEvaluation`.
  - **Windows of one unit are deprecated, not refused.** A bootstrap over units would draw that
    unit every time, so there is no interval to read from it. But `chc.decision` is in the stable
    tier, whose breaking changes wait for 1.0 and get a deprecation cycle, and refusing would fail
    a call that ran before. Until 1.0 such windows are read as before: `evaluate_plan` by
    `"pdis"`, the windows taken as independent, under an interval too narrow for them, with
    `bootstrap` and `versus_logger` `None` and no comparator. Each such call warns once with a
    `DeprecationWarning` saying so, that from 1.0 a panel of one unit raises `DecisionError`, and
    to evaluate on a panel of several units instead; it logs the same at `WARNING`
    (`chc_event="one_unit"`).
- **The logger beside the plan.** `PlanEvaluation.versus_logger` is a `LoggerComparison`: the
  logs' own mean cost over the horizon, which is the logger's value from the windows' starts; the
  plan's value less it; and that difference's interval, read off the difference within each draw.
- **Where it lives.** `Prescription.evaluate` is on the stable `chc.decision` surface, and the
  change is additive there: three keywords whose defaults are the corrected behaviour. The new
  types are in `chc.evaluation`, evolving, beside `PlanEvaluation`, whose two new fields are `None`
  from `evaluate_plan`. `evaluate_plan` takes its episodes as independent, as before.
- **Two costs of rerunning it.** The smoothing's grid, searched again on every draw, is read in one
  recursion batched over its levels, exact to the scalar one; and the one-step linearisation is
  compiled, once a model.

## Consequences

- `tests/test_target_trial.py` builds a market in which a zone leaves three periods after its
  supply first strays more than 1.5 from zero, three periods being the plan's horizon, and computes
  every truth on the market's own law from the starts of the windows each reading takes. On a panel
  of 4000 zones, at `model_error=0` so that each interval is its bootstrap's alone, the calendar's
  intervals cover the plan's value and its difference from the logger's, and the cut-back interval
  lies wholly below the plan's value on its own windows. Over twelve such panels, from seed 2000,
  where 84 to 86 % of the zones leave and the calendar reads 4119 to 4313 windows against the
  cut-back 6793 to 6930, each estimate's error over its standard error, the half-width over 1.96:

  | windows | `model_error` | plan's value covered | its error, mean (range) | difference covered | its error, mean (range) |
  |---|---|---|---|---|---|
  | calendar | 0 | 12 of 12 | +0.02 (−0.94 to +1.15) | 12 of 12 | −0.45 (−1.49 to +0.95) |
  | calendar | 1 | 12 of 12 | +0.12 (−0.45 to +0.62) | 12 of 12 | −0.03 (−0.86 to +0.51) |
  | cut back | 0 | 0 of 12 | −6.53 (−8.09 to −4.56) | 3 of 12 | +2.85 (+1.57 to +4.82) |
  | cut back | 1 | 3 of 12 | −2.32 (−2.84 to −1.58) | 11 of 12 | +1.33 (+0.49 to +2.16) |

  Before this change, `evaluate` cut back and read the delta method over windows. On the first
  three of those panels its interval missed the plan's value on its own windows by 6.5, 6.1 and 5.5
  of its standard errors, and the plan's value on the calendar's windows by more.
- `tests/test_evaluation.py` holds the bootstrap to independent routes:
  - **Units, not windows.** Sixty units, each one run of a logger whose slower mode keeps 98% of
    itself a step, cut into ten consecutive windows. Over 40 replicates the windows' own interval
    covered 28 and was 0.57 of the estimate's spread; the bootstrap over units covered 37 and was
    1.12 of it. The test asks at least 35, the binomial's 1.4% tail at 0.95.
  - **The cluster-robust sandwich.** With the plant, the logger and the smoothing held fixed, the
    bootstrap's half-widths over 150 units of four windows are the sandwich's, each episode's
    influence summed within its unit (Liang and Zeger 1986): at 4000 draws the plan's is 1.0% and
    the difference's 1.2% from it, where the test allows 6%. The difference's spread is 0.67 of the
    two values' spreads in quadrature.
  - **The whole analysis on each draw.** Replaying the draws from the seed and handing each drawn
    panel to `evaluate_plan`, with the plant fitted to it by least squares, gives both intervals
    to `1e-9`.
  - **The batched grid.** On 40 random cases each level's moment matches the whole-trajectory
    integral to `1e-9`, through the steps where it diverges; and on twelve, the level chosen is the
    one a search scoring each level alone chooses.
  - **Refusals.** At the certificate's edge, some draws are refused, counted and logged; when all
    but one are, the evaluation is refused.
- All 55 mutations tried fail a test: 21 of the bootstrap and the batched grid, 14 of the windows
  and of what `Prescription.evaluate` passes on, and 20 of the reading of one unit: its warning,
  the warning's message, its log and what it passes on. Earlier rounds left three survivors, and
  each got a test: the plant built once on every unit for every draw, which the replay test catches
  now that its plant is fitted to each drawn panel; the caller's time zero not passed on, which the
  dropout test catches by counting the windows each reading takes; and two units' windows read as
  one unit's, which the one-unit test catches by drawing two zones. That test also reads the
  warning's file, so a warning pointing into the library rather than at the caller fails it.
- **Few units.** A bootstrap over few units draws few distinct panels. In the coverage test's
  world, 600 windows split over 5, 10, 20 and 60 units, 40 replicates each, the bootstrap covered
  34, 36, 34 and 37, and the windows' own interval 21, 26, 20 and 28.
- **One unit is read as before, deprecated**: a single long run, one building or one geo, has
  nothing to resample by unit. `tests/test_target_trial.py` holds the reading of one zone's 1200
  windows to `evaluate`'s on this change's base, `fe39f1b`, to `1e-9`: with the smoothing chosen
  under a binding `min_effective`, and with the logger and the smoothing given. From 1.0 such a
  panel raises `DecisionError`; `evaluate_plan` still reads its windows as independent.
- An evaluation runs the analysis `n_resamples + 1` times.
- On a panel whose units are observed throughout, the calendar's windows are those cut back, and
  the value is unchanged; the interval is the bootstrap's.
- `scripts/bench_logger_check.py` reads `Prescription.evaluate`'s interval, and the coverage
  ADR 0028 records for it was read under the delta method.

## Not built

- **Weighting by the chance of staying.** A unit that leaves within `H` periods of a time zero
  because of what happened after it takes its window with it, and the windows left are not the
  trial's. Inverse probability of censoring weights would correct that (section 22.4 and chapter
  17); they need a model of leaving, and nothing here fits one.
- **A bootstrap over time within a unit**, blocks of consecutive windows, for a panel of one or a
  few long runs.
- **Other comparators**: another fixed plan, or no action, each its own weights on the same draws.
- **Percentile and bias-corrected intervals.** The normal interval keeps `evaluate_plan`'s form and
  the model's widening.

## Alternatives considered

- **Keeping the cut-back windows as the default.** Rejected: on the market above, its interval
  missed the plan's value on its own windows on every panel at `model_error=0`, and on nine of
  twelve with the model's correction counted.
- **Time zeros counted forward from the panel's first period.** Rejected: on a panel whose length
  is not a multiple of `H` it leaves out the most recent periods, which are the closest to
  deployment.
- **Every period a time zero**, overlapping windows. Rejected: each period is read up to `H` times,
  and the certificate counts windows as samples, so the effective sample size it certifies would
  be up to `H` times too large.
- **A cluster-robust sandwich for the interval.** Rejected: it holds the linearisation, the
  logger's fit and the smoothing at their values on every unit, which section 22.4's "bootstrapping
  the entire analysis" does not. The tests use it as the oracle where they are held.
- **Two marginal intervals for the difference.** Rejected: both values read the same draws, and
  what they share would be counted twice.
