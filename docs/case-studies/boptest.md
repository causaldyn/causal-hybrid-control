# A heat pump from a weather-compensated log

`bestest_hydronic_heat_pump` is BOPTEST's emulator of a single-zone house heated by a modulating
heat pump. Here `prescribe` drives it closed loop, planning from a log that a weather-compensated
controller left behind. That controller's outdoor-reset curve raised the modulation on cold days,
and cold days are the days the house loses heat fastest, so a model fitted on the log as it stands
mixes the heat pump's effect with the weather's.

This case study is not a script in this repository. It runs in
[`causaldyn-bench`](https://github.com/causaldyn/causaldyn-bench), for hours, against a live
emulator, and it was pre-registered: the module's docstring, committed before any scored episode.

```bash
just boptest-prescribe-live    # in causaldyn-bench, with the BOPTEST stack up
```

At every half-hour step both arms make the same call from the zone temperature just measured, and
the heat pump gets the first action of the schedule:

```python
prescribe(
    panel,
    levers=[Lever("modulation", 0, 1, unit_cost=0)],
    target=Target("zone", value=target),
    horizon=16,
    dt=0.5,
    tolerance=0.5,
    x0=[zone],
    adjustment=graph,
)  # the naive arm passes adjustment=()
```

The graph says what the logging controller and the house do: the reset curve reads the outdoor
temperature, its comfort feedback reads the bound, and occupancy, which nothing measures, sets both
the bound and the internal gains. `prescribe` resolves it to `{outdoor, solar, bound}`. The target
is the highest lower comfort bound BOPTEST forecasts over the plan's eight hours, plus an offset
swept over seven values from -0.5 to +1.5 K. The lever is free, so the offset is the only knob that
trades comfort for energy. Each arm's seven weeks trace a front, and the gate reads its energy on
that front at the discomfort of BOPTEST's own baseline controller in the same week, so neither arm
can win by heating less.

## What it found

Six replicates, each a 20-day log of the reset controller and the week after it, the weeks running
from day 20 to day 62 of the year. Energy is BOPTEST's `ener_tot` in kWh/m2, read on each arm's
front at the baseline's `tdis_tot` in K h, and `excess = E_naive / E_adjusted - 1`:

| replicate | baseline discomfort | baseline energy | adjusted energy | naive energy | excess |
|---|---|---|---|---|---|
| 0 | 5.1359 | 1.7260 | 1.5813 | 1.4436 | -0.0871 |
| 1 | 3.4514 | 1.8867 | -- | 1.4391 | -- |
| 2 | 5.9027 | 1.7812 | 1.5764 | 1.4143 | -0.1028 |
| 3 | 5.6058 | 1.6424 | 1.4856 | 1.3987 | -0.0585 |
| 4 | 4.2318 | 1.6473 | 1.4717 | 1.1515 | -0.2176 |
| 5 | 3.7969 | 1.4327 | 1.2583 | -- | -- |

The mean excess is -0.1165, with a 95% t-interval of [-0.2276, -0.0054] over the four replicates
where both arms have a reading. So the decision reads REFUTED: at the baseline's comfort the naive
call was the cheaper one, in every replicate that could be read. **The verdict does not stand**,
because both validity checks failed:

- V1: two fronts have no reading at the baseline's discomfort. In replicate 1 the adjusted arm's
  cheapest week was also free of discomfort, so its front is a single point more comfortable than
  the baseline. In replicate 5 the naive arm never got as comfortable as the baseline.
- V2: one naive week had 15.2% of its solves stopped by the iteration budget, against a limit of
  1%.

Beside the gate, not gated: at the baseline's discomfort the adjusted arm needed 10.45% less energy
than the baseline itself (95% t-interval [8.57%, 12.33%]) and the naive arm 21.12% less ([13.52%,
28.72%]), each over the five replicates where it has a reading, which are not the same five.

## Where the naive call breaks

Fitted on the log as it stands, the heat pump's channel is linear in the zone temperature, and it
changes sign inside the range a heating target can reach. In replicate 2 it is `3.241 - 0.1413 T`,
zero at 22.94 C. Above that temperature the model reads more heat as cooling, so once the zone is
above both the crossing and the target, the plan's answer to a room that is too warm is more heat.
The naive loop ran away in eight weeks, all in the three replicates whose crossing sat lowest (2, 4
and 5). Over those weeks it held mean modulations of 0.63 to 0.97 and used 2.35 to 3.75 kWh/m2
against the baseline's 1.43 to 1.78, and its discomfort reached 522 to 1403 K h. BOPTEST's
`tdis_tot` counts excursions above the upper comfort bound as well as below the lower one, and at
those modulations these were overheating. In replicate 5 it happened at every offset from -0.25 K
up, which is why that replicate's naive front never reaches the baseline. Track D-causal found the
same sign change on the same emulator, at 22.62 C on one of its seeds, where it pinned the command
at zero instead. The adjusted arm never ran away.

## What the certificate said, and what the plant did

Every step is one `prescribe` call and only its first action is applied, so the step ahead is the
part of each plan the plant can check. Over every week of an arm: the range of the trustworthy
prefix, the one-step figures pooled, the largest finite regret bound and the share of calls where
it was infinite, and the largest share of any one week's calls stopped by the iteration budget or
left at the zero guess:

| arm | trustworthy steps | tube one step ahead K | one-step error rms K | largest error K | inside the tube | inside the tolerance | regret bound | bound infinite | stopped by budget | no progress |
|---|---|---|---|---|---|---|---|---|---|---|
| adjusted | 1-1 | 0.2926 | 0.1305 | 0.7931 | 0.959 | 0.991 | 2.15e-06 | 0.034 | 0.000 | 0.193 |
| naive | 0-0 | -- | 0.3263 | 1.6513 | -- | 0.872 | 5.34e-05 | 0.121 | 0.152 | 0.247 |

Where they disagree:

- The adjusted certificate trusted one step on every call, with an error tube of 0.29 K on average.
  The plant's one-step error left that tube on 4.1% of steps, reaching 0.79 K. The tube prices the
  channel's standard error, which is a scale and not a coverage guarantee.
- The regret bound stayed small wherever it was finite, the runaway weeks included. It is a gap in
  the planning objective on the fitted model, not a statement about the plant. It was infinite, the
  objective not convex over the box, on 3.4% of the adjusted calls and 12.1% of the naive ones.
- Nothing in the certificate bounds comfort, which is what the gate matches on and what the runaway
  weeks lost: `prescribe` refuses a constraint on its target column, so there is no barrier and no
  `gamma*`.

Where they agree: the naive certificate trusted nothing, because an `asserted` identification never
earns a step whatever the model predicts, and the runaway weeks bore that out. They are also what
pushed the naive arm past the V2 limit.

Scope, and it bounds what the numbers mean:

- The verdict is void, so this page makes no claim about what adjustment costs or saves here. The
  four readings that exist all favour the naive call.
- The lever is priced at zero, so each plan minimises distance to the target, not energy. In
  replicate 4 at offset 0 the naive arm held a higher mean modulation than the adjusted one (0.296
  against 0.253) and used less energy (1.1529 against 1.4770 kWh/m2). Energy use here is not
  proportional to modulation, and neither arm was optimising what `ener_tot` measures.
- The façade could not state three things the problem has: a bound on the steered state, as above;
  a weather term in the drift, so the weather the adjusted arm partials out never enters its
  forecast; and a target that changes within the horizon, so the comfort schedule reaches the plan
  only through re-planning, which pre-heats up to one horizon early.
- Six replicates, sized to the emulator time available rather than for power. Neighbouring logs
  share 13 of their 20 days, so the interval treats replicates as more independent than they are.
- Every number is float64, from chc 0.5.1.

Results:
[`results/boptest_prescribe/results.md`](https://github.com/causaldyn/causaldyn-bench/blob/main/results/boptest_prescribe/results.md).
Code:
[`causaldyn_bench/boptest_prescribe.py`](https://github.com/causaldyn/causaldyn-bench/blob/main/src/causaldyn_bench/boptest_prescribe.py),
through [`prescribe`](../api/decision.md).
