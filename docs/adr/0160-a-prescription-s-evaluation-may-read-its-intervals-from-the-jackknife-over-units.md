# ADR 0160 — A prescription's evaluation may read its intervals from the jackknife over units

**Status:** accepted, 2026-11-21. Experimental. Amends ADR 0049 (a prescription evaluated as a
target trial) and ADR 0095 (the draws a bootstrap reads). The default, the bootstrap over units,
does not change, and no value or interval it reads moves.

## Context

`Prescription.evaluate` reads its intervals from a bootstrap over the units: every unit drawn with
replacement, the whole evaluation run again on each draw, the interval the estimate plus or minus
1.96 standard deviations of the draws (ADR 0049). With few units it covers short of 0.95. The
estimate of a quadratic cost under importance weights is skewed to the right, and the spread the
draws read rises and falls with it, so the misses fall where the estimate reads low (ADR 0095).

The shortfall was re-measured on the design where it was found: the two-state market of
`tests/test_evaluation.py` (a ramp plan, an affine logger of unit variance) and ADR 0083's scalar
unit (`a = 0.6`, an open-loop plan), 30 units of ten windows of five steps, each unit one run of
the logger from its stationary law, `min_effective = 10`, `model_error = 0`, 200 draws. Panel `k`
of a plant is drawn from `numpy.random.SeedSequence([20261013, 702 or 703, 30, 10]).spawn(k + 1)[k]`,
and the bootstrap's seed from the panel's stream after the data, as in the first measurement. The
first 500 panels reproduce that measurement to the bit on the current code. The truth is the
plan's expected cost over the horizon from the logger's stationary law, by moment propagation.
On 2000 panels a plant (Monte Carlo SE 0.005 to 0.007), each candidate read on the same panels and
the same draws:

| interval | market value | market difference | scalar value | scalar difference | width / needed |
|---|---|---|---|---|---|
| bootstrap, normal (the default) | 0.917 | 0.923 | 0.895 | 0.924 | 0.89–0.93 |
| bootstrap percentile | 0.909 | 0.915 | 0.893 | 0.919 | 0.87–0.90 |
| bootstrap, Student's `t(G - 1)` | 0.928 | 0.935 | 0.908 | 0.936 | 0.93–0.97 |
| cluster delta method, the fitted logger's term included | 0.919 | 0.927 | 0.897 | 0.913 | 0.89–0.92 |
| CR2 on it, `t(G - 1)` (Bell and McCaffrey 2002) | 0.934 | 0.941 | 0.907 | 0.929 | 0.94–0.98 |
| BCa, the jackknife's acceleration | 0.899 | 0.902 | 0.884 | 0.895 | 0.89–0.94 |
| Hall's (1992) transformation | 0.917 | 0.914 | 0.906 | 0.901 | 0.94–1.12 |
| bootstrap-t, equal-tailed | 0.927 | 0.922 | 0.909 | 0.916 | 1.00–1.08 |
| bootstrap-t, symmetric (Hall 1988) | 0.938 | 0.944 | 0.914 | 0.942 | 1.03–1.19 |
| wild cluster restricted bootstrap-t, Rademacher | 0.933 | 0.938 | 0.907 | 0.927 | 0.92–0.97 |
| the same, Webb's six points (Webb 2023) | 0.932 | 0.938 | 0.908 | 0.928 | 0.92–0.97 |
| **jackknife CV3, `t(G - 1)`** | **0.938** | **0.944** | **0.919** | **0.944** | **0.98–1.04** |

"Width / needed" is the mean width over twice 1.96 standard deviations of the estimate across
panels. The bootstrap-t studentises each draw by its own cluster delta-method standard error; the
wild bootstrap reads the units' linearised values, inverting the symmetric test at 0.05 with 999
weights a panel. On 750 panels a plant of 60 units of five windows, drawn the same way with the
tags 700 and 701 and reproduced to the bit, the default covered 0.932, 0.936, 0.900 and 0.933 in
the table's order, and the jackknife 0.943, 0.956, 0.924 and 0.948.

No candidate restores 0.95 for the plan's value on either plant: the market's best reads 2.2 Monte
Carlo SEs short, the scalar unit's 5.1. Its misses stay one-sided, the truth above the jackknife's
interval 6 to 9 times as often as below it: those are the panels that hold none of the heavy
weights carrying the upper tail, and no reading of such a panel alone sees them. No candidate
covers more than the jackknife on any cell; the symmetric bootstrap-t ties it on the market's two
and is 4 % to 16 % wider.

## Decision

- **`Prescription.evaluate(interval="jackknife")`**, opt-in; `"bootstrap"`, the default, is the
  reading it was. Each unit is left out in turn with all of its windows, and the evaluation runs
  again on the rest as on a draw: the plant, the logger unless given, the initial law, the
  smoothing, the certificate and both values. The spread is CV3 as MacKinnon, Nielsen and Webb
  (*J. Econometrics* 232, 2023) write it, `sqrt((G - 1)/G sum_g (v_g - v)^2)` about the estimate
  `v`, and each interval is Student's `t` with `G - 1` degrees of freedom times it, widened by the
  model's correction as the bootstrap's is: what they advise with few clusters. It is at least the
  jackknife's variance about the leave-outs' mean, which Efron and Stein (*Ann. Statist.* 9, 1981)
  prove conservative in expectation for a statistic symmetric in independent units; on the panels
  above the two read coverages within 0.001 of each other.
- **Each leave-out is certified against `min_effective` scaled to the windows it keeps**, so the
  certificate asks of it the effective share it asked of the panel. Unscaled, a panel near its
  edge would see every leave-out refused for being one unit smaller. A leave-out the certificate
  still refuses refuses the evaluation (`InfeasibleEvaluation`, naming the unit): the jackknife
  reads every leave-out, and one left out of it is no jackknife.
- **The record says which.** `degrees_of_freedom` is `G - 1`, `bootstrap` is None, and
  `settings.interval` holds the keyword, a new key, so `EVALUATION_SCHEMA_VERSION` stays 1 (ADR
  0145). `n_resamples`, `seed` and `max_refused` are not read.
- **The docstring states the scope**: the coverage at 30 and 60 units, for both readings.

## Consequences

- Under the jackknife the intervals cover 0.938 to 0.944 where the bootstrap's covered 0.917 to
  0.924 on the market and the scalar unit's difference, and 0.919 against 0.895 for the scalar
  unit's value, 1.08 to 1.13 times as wide. The shortfall below 0.95 narrows by 44 % to 78 % and
  is not closed.
- It runs `G` evaluations against the bootstrap's `n_resamples`: fewer at 30 units than the
  default 200 draws, more past 200 units.
- On ADR 0049's persistent loop (60 units of ten windows of five steps, a logger whose loop keeps
  98 % of its slower mode a step, an open-loop plan, `min_effective = 100`), 1000 panels from
  `SeedSequence([20261121, 60, 10])`, the plan's interval covered 0.913 (MC SE 0.009) by the
  bootstrap and 0.923 (0.008) by the jackknife, 0.99 and 1.03 of the width needed: short there
  too, where ADR 0049's 40 replicates read 37, and again the jackknife covers more.
- `tests/test_evaluation.py` replays the jackknife by hand through `evaluate_plan` with a plant
  fitted to each leave-out, and holds the scaled certificate and the refusal at the edge; on 40
  panels of the table its bounds agree with the measurement's to `5e-12`.

## Alternatives

- **Make the jackknife the default.** It moves every interval the default reads, and it does not
  reach the nominal level either; opt-in first.
- **Any other reading in the table.** Each covers less on some cell, or as much at a greater
  width.
- **Correct the centre by the jackknife's bias estimate.** On the same panels it covered 0.873 to
  0.941 at 60 units and 0.882 to 0.934 at 30: the correction's noise costs more than the bias it
  removes.
- **A double bootstrap, calibrating the level on draws within draws.** Forty thousand evaluations
  a panel, and it reads the same panel: where the panel lacks the upper tail, so do its draws'
  draws.
