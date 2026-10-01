# ADR 0041 — Intervals that carry the discounts' uncertainty

**Status:** proposed, 2026-10-02. Experimental, as `chc.dlm` is.

## Context

ADR 0022 measured what the discount DLM's intervals cover on `scripts/bench_dlm.py`'s random-walk
world. With the discounts picked by one-step likelihood on a 6 x 6 grid, a channel's contribution
over 13 weeks was covered 0.782 and 0.770 of the time at a nominal 0.90; averaged over the grid by
likelihood (West and Harrison's multi-process class I), 0.830 and 0.826. It left to the module's
users what carries the discounts' uncertainty into an interval, and every return a media-mix model
reports is such an interval.

A pilot on fresh series found where the loss is. On series drawn from the discount model itself,
the likelihood's pick covered 0.87 and 0.95. On the random walk, a law with a fixed evolution
variance, which is that world's own, covered 0.89 and 0.86 at the true variances and 0.78 and 0.77
at its likelihood pick, as the discount law does. So the loss is the selection, not the law. Over
156 steps an evolution variance is weakly identified, the likelihood is flat across a range of
them, and its best point is a selection whose interval ignores the range. A mixture weighted by
the same likelihood leans on the same point.

## Decision

- **`confidence_set(fits, parameters, level=0.95)`** in `chc.dlm`: the fits a likelihood-ratio
  test at `level` does not reject, those within `chi2_parameters(level) / 2` of the best
  log-likelihood, in the order given. `parameters` is how many settings the fits vary over. The
  caller states it: a grid that moves two blocks' discounts together varies one.
- **An interval is the union of the members' intervals** (projection; Berger and Boos 1994), from
  the lowest lower end to the highest upper end. When the set holds the discounts under which an
  interval would be right, the union holds that interval. The module returns the set and the
  caller takes the union over what it reports, since a window's total, a change and a return are
  each a different functional of the draws.
- **Refused:** no fits; fits of other observations than the first's, whose likelihoods a ratio
  does not compare; a `parameters` that is not a positive integer; a `level` outside `(0, 1)`.
- Each set is logged (`chc_event="dlm_confidence_set"`) with its size and its margin.

## Consequences

Pre-registered in `scripts/bench_dlm.py`'s docstring (`b1e20c0`) before the scored run: each
channel's 13-week contribution covered in at least 0.88 of 500 series, both on the random walk
(the series of ADR 0022's rows, which the run reproduced exactly) and on 500 series of the
discount model itself at 0.9 and 0.9 with an explicit state path. At a nominal 0.90, with a Monte
Carlo standard error of 0.013:

| | random walk | discount world |
|---|---|---|
| the likelihood's pick | 0.782, 0.770 | 0.852, 0.848 |
| the class I mixture | 0.830, 0.826 | |
| at the true discounts | | 0.902, 0.892 |
| projected | 0.938, 0.944 | 0.940, 0.946 |

- **The gate is met in both worlds.** Predicted: 0.91 and 0.93, 0.93 and 0.99.
- **It over-covers**, 0.94 where 0.90 was asked: a union is conservative by construction. The
  price is width, in medians 1.44 times the pick's interval on the random walk and 1.21 times the
  interval at the true discounts on the discount world. The set held a median 9 of the 36 pairs on
  the random walk.
- **The pick under-covers on the model's own world too**, 0.85: selection costs coverage wherever
  the likelihood is flat, not only under a wrong law.
- **No refits.** The members are fits the choice already made, and an interval costs one backward
  sample per member.

## Alternatives

- **The class I mixture** (ADR 0022's averaged row): 0.830 and 0.826. Its weights are the
  likelihood's, which favours the high discounts the selection favours.
- **The pick's interval recalibrated by simulation from the fitted model** (Beran's prepivoting,
  40 simulated series per fit): 0.84 and 0.84 on the pilot, at the cost of refitting the whole grid
  for every simulated series.
- **Discounts chosen by the 13-step-ahead score**, the horizon of the reported window: 0.73 and
  0.71 on the pilot.
- **The set's draws pooled** in place of the union, a flat mixture over the set: 0.79 and 0.81 on
  the pilot.
- **A fixed evolution variance as a second law, chosen with the discounts.** The pilot found the
  law was not the loss.

## Not built

- **A union helper.** The union is two reductions over what the caller reports.
- **Coverage beyond the two worlds:** other discounts, lengths and numbers of blocks, and finer
  grids, whose sets are larger and whose unions are wider.
