# ADR 0020 — An internal pilot for a switchback

**Status:** proposed, 2026-09-29.

## Context

The switchback plan of ADR 0012 quotes each effect's standard error and minimum detectable effect
from the working model: a first-order state per zone, and zones independent of each other. The zone
market of `chc.zones` breaks both assumptions. Its incentive also acts through a stock, and half of
what a zone recruits comes from its neighbours. There the model-free readings stay within 0.34% of
the truth, but the readings of a horizon or of the steady state spread further than planned:

- 1.04 times for `tau_2`'s local projection;
- 1.06 times for `tau_5`'s;
- 1.64 to 1.73 times for the steady state's block difference.

A test at the planned MDE then rejects 0.43–0.46 of the time for the steady state, against a
planned power of 0.80 (`scripts/bench_switchback.py zones`).

The run's first periods hold the spread the plan missed. Two things must be settled to use them:

- how to carry a pilot's standard error to the whole run;
- what multiplier turns that standard error into an MDE with the plan's power.

## Decision

- **`restate_mde(plan, estimand, lever, outcome)`** reads `estimand` off the first periods of its
  arm, with the plan's own analysis and design. It returns the plan's `EstimandReport` with the
  standard error and MDE restated for the whole arm. The bias and the loss stay the plan's, because
  the design has not changed.
- **The test at the end is unchanged, and so is its level.** The MDE is restated, and the run is
  not resized.
- **A regression's standard error is carried by its rows:**
  `se_run = se_pilot sqrt((n_pilot − L) / (n_run − L))`. Here `L` is the number of lags the
  regression's covariance keeps, which is also the number of rows it sheds:
  - 0 for the plug-in's regression;
  - 1 for IV;
  - `H − 1` for the local projection.
- **The block difference's standard error is carried by its variance factor.** The factor has four
  parts (`validation/switchback_pilot.mac` STEP 1):
  - A centred block's score has variance `1 + (1/on + 1/off) / 4` times a block's own.
  - The blocks before a zone has shown both settings are lost.
  - A zone's difference averages its centred blocks.
  - The zones with at least two centred blocks are weighted alike.

  The run's later blocks are counted at their expected settings, `on + j/2` and `off + j/2`
  (STEP 4).
- **The multiplier is the noncentral t's:** `MDE = c se + |bias|`.
  - `c` is the `power` quantile of the noncentral t with `nu` degrees of freedom and noncentrality
    `z_(1 − alpha/2)`.
  - When the pilot's variance is chi-square on `nu`, the run's power at that MDE, averaged over the
    pilot, is the plan's power. STEP 3 integrates it to 0.8 within 3e−15 at `nu` = 8, 7.2 and 36.5.
  - For a regression, `nu = zones (n − L) / (2L + 1)`.
  - For the block difference, `nu` is Satterthwaite's over the blocks, because its standard error
    squares one sum over the zones for each block.
- **`SwitchbackPlan` keeps `alpha` and `power`,** so the restatement uses the plan's own. The module
  has not been released, so the new fields break no caller.
- **Refusals:**
  - the plan does not read the estimand;
  - the pilot covers other zones than the plan, or is not shorter than the arm;
  - a block pilot has fewer than ten blocks a zone;
  - a zone showed one setting in all of its pilot blocks, so when its blocks will be centred is
    unknown;
  - anything `read_switchback` refuses.
- **Top level.** The lifecycle page files `restate_mde` under Experiment, so `restate_mde` and the
  `SwitchbackPlan` it takes are in `chc.__all__` (ADR 0016).
- **Experimental,** like the rest of the module.

## Consequences

Measured by `scripts/bench_switchback.py pilots` on `chc.zones`' market: 4000 runs of each plan
without a state model, under linear and harmonic matching. Two powers are reported, both for the
run at the restated MDE:

- against the truth;
- about the reading's own mean, which leaves out the reading's own bias.

No number here is a timing.

| reading | pilot | refused | restated SE / spread | power | about the mean |
|---|---|---|---|---|---|
| channel, regression | a tenth, quarter, half of 2000 | 0 | 1.013–1.017 | 0.796–0.799 | 0.806–0.808 |
| `tau_2`, local projection | a tenth, quarter, half of 2000 | 0 | 1.014–1.017 | 0.799–0.805 | 0.803–0.809 |
| `tau_5`, local projection | a tenth, quarter, half of 2000 | 0 | 0.997–1.011 | 0.761–0.781 | 0.780–0.794 |
| steady state, blocks of 50 | 10 of 40 blocks | 49 / 49 | 0.960 / 0.962 | 0.787 / 0.787 | 0.751 / 0.754 |
| | 15 blocks | 1 / 1 | 0.990 / 0.987 | 0.817 / 0.824 | 0.786 / 0.791 |
| | 20 blocks | 0 / 0 | 1.003 / 0.999 | 0.834 / 0.837 | 0.802 / 0.803 |

At the planned MDE, the same runs rejected:

- 0.794–0.796 for the channel;
- 0.773–0.778 for `tau_2`;
- 0.731–0.737 for `tau_5`;
- 0.421 and 0.459 for the steady state.

- **The pilot restores the steady state's power,** from 0.42–0.46 to 0.79–0.84 against the truth.
  About the reading's own mean, it is 0.75 at ten blocks, 0.79 at fifteen and 0.80 at twenty.
- **The ten-block shortfall comes from spillover.** Two zones' scores for the same block covary by
  `c12 c21 / 4` whatever the centring holds (STEP 2). The centring therefore does not inflate the
  cross-zone part, but the ratio treats it as if it did.
  - Test ring (`tests/test_switchback.py`), where the neighbours' coins make up most of the block
    difference's variance and the plan's SE is 0.29 of the spread. The ratio read 0.934 of the
    spread at ten blocks, with power 0.737, and 0.982 at twenty, with power 0.793.
  - Same ring without spillover: 0.991 at ten blocks, with power 0.792.
- **`tau_5`'s shortfall is its own bias.** About the reading's own mean its power is 0.78–0.79. The
  local projection is biased by 0.05 of a spread, and that bias belongs to the reading, not to the
  pilot.
- **The steady state's power against the truth runs high** because the plan's bias allowance is
  larger than the reading's actual bias there.
- **A ten-block pilot was refused on 1.2% of runs,** each time for a zone that had shown one setting
  in all ten blocks.
- **The multiplier is what holds the power at short pilots.** On the working model at the
  flagship's persistence (`a = 0.87`, four zones, blocks of 50 with a washout of 21):

  | pilot blocks | `nu` | power, `z + z_power` | power, noncentral t |
  |---|---|---|---|
  | 5 | 2.7 | 0.589 | 0.735 |
  | 10 | 7.2 | 0.711 | 0.792 |
  | 20 | 17.5 | 0.773 | 0.806 |

- **The design's `nu` is an approximation.** The local projection for `tau_5` read an empirical 51
  against the design's `n/9 = 87`. `DIM(5, 4)` at a tenth read 13.7 against Satterthwaite's 16.9.
  At those values the multiplier moves by 0.7% and 1.2%.
- **At a tenth of the arm, two readings of the tests' model are off for reasons of their own:**
  - IV's pilot has about 45 switches and read `tau_5`'s standard error 14% high, so power was
    0.835;
  - the plug-in's steady state carries its O(1/T) bias, so power was 0.764 against the truth.

  At a quarter of the arm, every simulated plan is within 0.03 of 0.80, and a test asserts it.
- **Checked:**
  - `validation/switchback_pilot.mac` runs STEPs 1–4.
  - `tests/test_switchback.py` holds the pilot's tests:
    - a transcription of the block factor;
    - the regressions' formula at two levels and powers;
    - the block path's wiring, with the plan's bias;
    - the power by Monte Carlo on every simulated plan and on two rings;
    - the refusals;
    - the log.
  - 33 mutations of the restatement and the block factor, all caught. The first pass left four
    alive, because Monte Carlo cannot see them: IV's lag, the zones in a regression's `nu`, and the
    two zone counts that leave out a zone centred less than twice. A deterministic test now holds
    each of them.

## Alternatives considered

- **Carrying the block difference by its periods, as for a regression.** Rejected: it counts the
  lost blocks and the thinly centred ones as data. On the working model, a pilot of ten blocks read
  1.195 times the run's SE, with power 0.849.
- **`c = z_(1 − alpha/2) + z_power`.** Rejected: at the degrees of freedom a block pilot has, it
  falls short of the plan's power. It gave 0.711 at ten blocks and 0.589 at five, and STEP 3
  integrates it to 0.731 at `nu = 8`.
- **Splitting the block factor into an own part and a cross-zone part, and inflating only the own
  part.** Rejected. It removes the ratio's bias under spillover, but reads each part off few
  blocks:
  - on the ring at ten blocks, power 0.751 against the ratio's 0.737;
  - on the working model, 0.772 against 0.792;
  - on the flagship, 0.788 against 0.787.

  It also refused more pilots (53 against 16 of 2000 on the ring), because its cross part can come
  out negative. Floored at zero, it overshot, reaching 0.850 on the flagship.
- **The exact binomial expectation of the later blocks' factors.** Rejected: it would sum over
  every later block's binomial, yet moves a zone's factor by under 0.2% on a balanced pilot and
  under 1% on one that showed a setting once or twice (STEP 4).
- **A floor of twenty blocks.** Rejected: the flagship's steady-state arm is forty blocks long, so
  the pilot would be half the run. Ten blocks holds the power where the zones are independent, and
  the shortfall under spillover is documented.
- **Resizing the run from the pilot** (Wittes and Brittain 1990). Not built: a switchback's length
  is usually its budget, and a resized run changes what the final test's level rests on. The
  restated MDE tells the operator whether the run can detect the effect. Whether to extend the run
  is the operator's call.
