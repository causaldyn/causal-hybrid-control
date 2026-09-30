# ADR 0025 — Randomisation inference for a switchback

**Status:** proposed, 2026-09-30.

## Context

`read_switchback` reads an effect with an interval that is asymptotic in `T`: Wald's for every
reading of a finite horizon, and Fieller's for the plug-in's steady state. A switchback is often
short. It may have a few blocks a zone, or a strong carryover, and there the Wald interval does not
hold its level. On the working model at `a = 0.8`, the block difference's Wald interval excluded a
true zero effect in 17.2% of the runs over 6 blocks, 11.9% over 8 and 7.2% over 12
(`scripts/bench_switchback_randomisation.py size`).

The design randomises the lever: a `MarkovDesign` draws each zone's path, and a `BlockDesign` one
fair coin a block. The design's own randomisation therefore gives an exact test (Fisher's), with no
appeal to `T`. Two things stand in the way:

- **Which schedules are the reference.** Some schedules cannot be read at all: a block schedule
  with fewer than two blocks after both settings, or a lever that never switches. The observed
  schedule is always one the reading read.
- **A sharp null under carryover.** A null that names `tau_H` alone does not fix the readings the
  lever leaves when it is off. The schedule moves each reading through the lever's whole path,
  `b sum_{s>=1} a^(s-1) u_(t-s)`, and two memories with the same `tau_H` impute different readings.

## Decision

- **`randomisation_test(lever, outcome, estimand, analysis, design)`** tests the sharp null that
  the lever moves no reading, in any zone and at any lag. It is *experimental*.
  - **The statistic** is the reading's `estimate / se`, as `read_switchback` reads it. It is
    computed for a batch of schedules at once through the reading's own cores (`_fit`,
    `_plant_fit`, `_projection_fit`, `_block_reading`). Studentised, it also keeps its level in
    large samples under the weaker null of an effect that averages zero (Wu and Ding 2021).
  - **The reference** is every schedule of a block design when there are at most 2^14 over the
    zones, all equally likely. Otherwise it is `draws` schedules drawn from the design, and the
    p-value is `(1 + at least as extreme) / (1 + read)`.
  - **Only the schedules the reading reads count.** Under the null the outcome is fixed, and so is
    which schedules the reading reads. The observed schedule, which it read, is a draw from the
    design restricted to them, so the p-value is exact given that the data were read. The
    statistic is NaN exactly where `read_switchback` refuses a schedule. That covers the local
    projection's refusal of a lever that is not i.i.d., at `5/sqrt(n)`, on every drawn schedule.
  - **It refuses** a design that cannot reject, before reading anything. `K` fair coins give a
    smallest p-value of `2^(1-K)` two-sided and `2^-K` one-sided, so at 5% a block design needs six
    coins over the zones two-sided and five one-sided. It also refuses too few draws, a lever the
    design cannot have drawn (a change inside a block; a Markov design that never or always flips),
    and the block difference of a Markov design.
- **`randomisation_interval(..., persistence)`** inverts the test under the working model's joint
  null `(a, b)`. It is *experimental*.
  - The joint null imputes the readings with the lever off, `y - b sum_{s>=1} a^(s-1) u_(t-s)`,
    through `scipy.signal.lfilter`, and the schedule cannot move them.
  - The interval holds every `tau_H = b S_H(a)` that some `a` in `persistence` leaves unrejected,
    at 21 memories spread evenly across the range.
  - Every null reads the same schedules: kept in memory up to 2^23 lever periods, drawn again past
    that.
  - The ends are searched from the first effect accepted within eight standard errors of the
    estimate. Steps double until the test rejects, then bisection runs to a thousandth of a
    standard error, and each end is reported on its rejected side. An end past 1024 standard
    errors is infinite.
  - When every effect within eight standard errors is rejected at every memory, it raises: the data
    are at odds with a first-order plant over the range, at least near the estimate.
- **`validation/switchback_randomisation.mac`** derives the imputation:
  - STEP 1: the kernel `F_a(u)` and the lever-off readings;
  - STEP 2: the block difference's carry `c_d = tau_H a^(dH)` from `d` blocks back;
  - STEP 3: `S_H(a) > 0` on `(0, 1)`, so `b = tau_H / S_H(a)` is defined;
  - STEP 4: the smallest p-values, 2^(1-K) and 2^-K, and the five- and six-coin thresholds.
- **Top level.** The lifecycle page files both functions under Experiment, beside
  `read_switchback`. By ADR 0016's rule they join `chc.__all__`, together with `MarkovDesign`,
  which the calls take. `RandomisationTest` is what the test returns, so it stays at
  `chc.switchback`.
- **No new dependency.** The test is numpy; the kernel is scipy's `lfilter`.

## Consequences

Measured by `scripts/bench_switchback_randomisation.py` on the working model, from a snapshot of
the library; `JAX_PLATFORMS=cpu`, rates only, no timing. The evidence is in
`~/.cache/chc-scratch/2026-09-30/ex5b/`.

- **Size** (`size`, `b = 0`, 1000 runs a case, less those the reading refused). At 5%:
  - The block difference of `tau_5` over blocks of 5 keeping 1, `a = 0.8`, over 6, 8 and 12
    blocks: the test rejected 3.6%, 5.2% and 4.6%, against Wald's 17.2%, 11.9% and 7.2%. The
    reading refused 115, 31 and 3 runs. That is the `2^(3-K)` share of one zone's `K` coins with
    fewer than two blocks after both settings: 12.5%, 3.1% and 0.2%.
  - The plug-in's `tau_3` over 30 and 60 periods of `MarkovDesign(0.3)`: 5.8% and 5.4%, against
    8.7% and 6.1%.
  - The steady state over 60 periods at `a = 0.9`: 4.4%, against Fieller's 5.3%.
- **Coverage** (`coverage` and `memory`, `b = 1`, 400 runs a case and 1000 for the long memory). A
  run whose every effect the test rejected counts as a miss.

  | case | Wald / Fieller | over the range | at the true memory | without memory |
  |---|---|---|---|---|
  | `tau_5` by blocks, `a = 0.8`, 8 blocks | 0.886 | 0.995, 78% unbounded | 0.956, 28% unbounded | 0.966 |
  | 12 blocks | 0.932 | 0.980, 10% unbounded | 0.935 | 0.950 |
  | 20 blocks | 0.932 | 0.973 | 0.948 | 0.953 |
  | plug-in `tau_3`, `T = 40` | 0.925 | 0.990 | 0.953 | 0.532 |
  | plug-in steady state, `a = 0.9`, `T = 60` | 0.890, 23% unbounded | 0.993 | 0.938 | 0 (41% reject every effect) |
  | `T = 120` | 0.918, 2% unbounded | 1.000 | 0.940 | 0 (48% reject every effect) |
  | `tau_2` by blocks of 2, `a = 0.95`, 12 blocks | 0.919 | 0.986, 95% unbounded | 0.949, 94% unbounded | 0.955 |
  | 24 blocks | 0.934 | 0.987, 93% unbounded | 0.947, 84% unbounded | 0.950 |

  The ranges are `(0.6, 0.9)`, `(0.8, 0.95)` and `(0.9, 0.97)`.
  - Where Wald failed (8 blocks, the steady state at 60 periods, the long memory), the range
    covered at least 0.986 and the true memory 0.938 to 0.956.
  - The true memory's coverage pooled to 0.946 over the eight cases' 4382 runs. At the truth
    itself, the test rejected 4.0% and 4.9% of 4000 runs over 8 and 12 blocks. The rest of the gap
    to 0.95 is the search's, or noise.
  - The price is width. At few blocks and long memory the exact interval is mostly unbounded, which
    is what the data allow and not a failure. The range's median width was 1.35 to 1.9 times
    Wald's where both were finite. At the true memory, the steady state's interval was half as
    wide as Fieller's, which pays for not knowing `a`.
- **The memoryless null is not a substitute.** It names `tau_H` and nothing else. For the block
  difference it covered 0.950 to 0.966, but that is measured and not proved. For the plug-in it
  missed: 0.532 of `tau_3`, and none of the steady states. So `persistence` has no default.
- **An oracle.** pyfixest 0.60.0's `ritest` ran in an ephemeral environment, on 200 logs of an
  i.i.d. lever over 80 periods, `y1 ~ u + y0`, with HC1 errors.
  - The channel's statistic agreed with its t to 4e-15 absolute.
  - Its p-values come from 999 permutations of the lever; this redraws 4999 schedules. They differed
    by 0.008 on average and 0.054 at most, correlation 0.9993, which the permutations' own noise
    allows.
  - Both rejected 3-4% of the 100 null logs, and 64-65% of the 100 logs with `b = 0.5`.
- **The reference set matters at few blocks.** Counting the unread schedules as never extreme is
  exact only over every schedule. Given a reading, it allows up to `alpha` over the share read. On
  six blocks at 10% that is 6 of 56 read schedules, 0.107, where the reference of the read ones
  rejects 4 of 56. At 5% the two agree on every case above, since the p-values move in steps of
  two schedules. The test that pins the change fails on the counting rule.
- **Tests** (`tests/test_switchback_randomisation.py`, 41):
  - the enumerated p-value is the share of the 248 read schedules of eight blocks, one by one;
  - a drawn p-value is uniform on `k/100`;
  - the batch equals `read_switchback` schedule by schedule, NaN where it refuses;
  - the complement negates the statistic;
  - every refusal;
  - the interval excludes 0 exactly when the test rejects;
  - coverage on ten blocks;
  - the memoryless null missing a horizon the joint null covers.

  Six mutations of the module were caught: the `+1` of the drawn p-value, the adjustment's sign,
  the one-sided resolution, an unstudentised statistic, the memory dropped from the kernel, and the
  enumeration's coin index.

## Not built

- **Bojinov, Simchi-Levi and Zhao's Horvitz–Thompson estimator** on an FIR plant of order `m`,
  which shares `tau_(m+1)` with `block_dim` at `H = m + 1`: the comparison on their ground,
  with their estimator as the oracle beside this one.
- **Coverage of the IV and local-projection intervals.** Their statistics and refusals are tested;
  their coverage is not measured.
- **Spillover in the interval.** The test's null of no effect anywhere holds under spillover. The
  interval's joint null imputes each zone from its own lever only.
- **The weak null's level** (Wu and Ding), which the studentised statistic keeps asymptotically, is
  not measured here.

## Alternatives considered

- **The unstudentised difference as the statistic.** Rejected, on an unscored exploration of 60
  runs a case (`tau_5` by blocks, `a = 0.8`). Over 8 blocks, 29 of 55 of its intervals at the true
  memory were unbounded, against 16 for the studentised one. Over 12 blocks it covered 0.881 of 59
  at a median width of 6.48, against 0.932 at 5.64. It also loses the weak null.
- **A null on `tau_H` alone, the memoryless imputation.** Rejected as the default, because it is
  not sharp under carryover and the plug-in's coverage falls to 0.53 and to 0. `persistence = (0, 0)`
  still asks for it.
- **Counting the refused schedules as never extreme.** Rejected, as above: that is not exact given
  a reading.
- **Permuting the lever, holding its count, as `ritest` does.** Rejected as the reference. A
  permutation is valid, but it is not the design: a `BlockDesign` redraws its coins, and the count
  is not fixed. The oracle shows the two agree within noise on an i.i.d. lever.
- **A bootstrap interval.** Rejected: it is asymptotic too, and at six to eight blocks it resamples
  from almost nothing.
