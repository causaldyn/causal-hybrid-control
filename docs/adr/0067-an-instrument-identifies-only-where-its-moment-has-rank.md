# ADR 0067 — An instrument identifies the channel only where its moment has rank

**Status:** accepted, 2026-10-07.

## Context

`fit_causal_residual(instrument=...)` reads the channel off the moment `Z'(y_res - D c) = 0`: `D`
the action residuals on the channel's features, and `Z` the first stage's fitted actions, on the
polynomial features of the state and the instrument, under the ridge, centred, on the same
features. It set `identified=True` because an instrument was named. Nothing checked that the
instrument moves the actions. Where it does not, the moment holds at every channel along the
direction it misses, and the ridge on `Z'D` and the moment's noise set the channel there. The
`unmoved` directions (ADR 0054) read the actions' own movement, not the instrument's.

An external review reproduced it with an instrument of zeros, the table's first row. The tests'
logs below it read the same way: identified, with an error the sandwich computed as if the channel
were identified.

| log | channel | truth | `channel_error` |
|---|---|---|---|
| the review's, 200 rows, no state, nuisance of degree 0 | 0.0 | 0.8 | 0.0 |
| the reference plant, 2000 rows, instrument of zeros, affine channel | `[0.81, -0.06]` | `[1.0, 0.5]` | 6.0 |
| the same, constant channel, first stage linear | `[-0.43, -0.13]` | `[1.0, 0.5]` | 0.35 |
| the same, affine channel, beside the confounder adjusted for | `[1.12, -0.76]` | `[1.0, 0.5]` | 47.6 |
| the plant, a column of noise less its projection on the state and the action | `[1.28, -0.53]` | `[1.0, 0.5]` | 321 |
| two levers, two instruments that move the first alone | `[0.79, -1.51]` | `[0.8, -0.4]` | 6.7 |
| one lever, its part the instrument would explain taken out | 1.19 | 1.0 | 4.8 |
| one lever, a weight of zero on every row the instrument moves | -0.85 | 1.0 | 1.1 |

The third row is the danger: an error of 0.35 looks like a weak instrument's, not like none. At
`nuisance_degree=0` the first stage is a constant, which no instrument enters, so the plant's own
instrument read a channel within 4e-9 of zero there, with an error of 2.3e-10.

## Decision

- **The relevance.** The fit reads how far the instrument moves each direction of the channel the
  log's actions move, the complement of `unmoved`'s span, which is where the moment is solved. It
  reads the canonical correlations between the first stage's push on the actions and the actions,
  each less its least-squares projection on the nuisance's features, on the channel's features,
  each row weighed as the moment weighs it. The push is the first stage's fit on the fit's own
  features of the state and the instrument, standardised as the nuisance's are, without the
  ridge. The correlations are the singular values of the unregularised moment with each side
  whitened. So they do not move with the units of the instrument, the actions or the state, and
  their count above zero is the moment's rank. For one action, a constant channel and no
  covariate beyond the state, they are the canonical correlations Cragg and Donald's statistic
  reads, and the one correlation is the root of Shea's partial `R^2`.
- **Rounding.** A push of at most 64 eps of the raw actions' size along a direction is none: the
  library's rule for a spread. So is a correlation of at most 64 eps. A zero instrument's push
  reached 6 eps at most, over 200 to 100000 rows, nuisance degrees 1 to 3 and channel degrees 0
  and 1.
- **Short of rank, the fit is not identified.** It reads as a fit with no adjustment and no
  instrument: `identified=False`, `method="observational"`, the channel of the action residuals as
  their own instrument, kept to compare, and no `channel_error`, `drift_error` or influence.
- **Beside `adjust_for` as well.** Naming an instrument says the covariates leave a confounder,
  so the fit does not fall back on them.
- **The directions the log never moves stay `unmoved`'s**, as under an adjustment set: the rank
  counts the directions the moment is solved on. A lever the log never used leaves the instrument
  identified where it moves the others.
- **Where the log moves no direction, the instrument identifies none.** The moment is then solved
  on nothing, and the fit is not identified. An adjusted fit of the same log keeps its flag, with
  every direction in `unmoved` (ADR 0054): its identification is the covariates' assumption, which
  no log can check, where an instrument's is a rank the log can.
- **Exposed, experimental.** `CausalDynamicsFit.instrument_relevance` holds the correlations,
  `(p,)` for the `p` directions a state the log moves, largest first; `instrument_rank` counts
  those above 0. Both are None without an instrument.

## Consequences

- *Each log in the table reads `identified=False`*, its channel the observational fit's. Beside
  the confounder adjusted for, that channel is the adjusted one, `[1.00, 0.50]`, and still not
  identified.
- *A relevant instrument fits as before, bit for bit*: on the reference plant at 2000 rows, the
  channel, the drift and both errors, under `euler` and `rk4`. Its correlations read 0.46, 0.43
  and 0.36.
- *The rank does not grade an instrument; the relevance does.* A column of noise drawn apart from
  the action keeps the rank. Over eight draws at each of 500, 2000 and 8000 rows on the reference
  plant, every fit read identified, the median channel off the truth by 1.1 to 2.7. The medians of
  the smallest correlation were 0.002 to 0.035, at most 0.075, where the plant's instrument read
  0.39 to 0.48.
- *The relevance tracks an instrument's gain on the lever* as the partial correlation in closed
  form does: to 5.6e-12 of itself at a gain of 1e-4. Near a right angle the push's rounding turns
  it by the rounding's share of the push, 3.7e-3 of itself at a gain of 1e-8, and the rank holds.
- *At `nuisance_degree=0` no instrument has rank*, and the fit says so where it read a channel
  within 4e-9 of zero.
- *Not covered*: inference robust to a weak instrument, such as Anderson and Rubin's test. A weak
  instrument's `channel_error` is a sandwich that weakness makes unreliable.
  `solve_channel_moment`, which takes the caller's own instrument, checks no rank.
- *Cost*: a least-squares fit for each side, a thin SVD of `N` by `p` for each, and one of `p` by
  `p`; little beside the error's reverse pass.
- *Tests*: an instrument of zeros under each channel degree, `rk4` and an adjustment set, and on
  200 rows under a nuisance of degree 3, whose push of zeros rounds to 6 eps; one orthogonal to the
  action; two instruments that move one of two levers, and both; a lever the log never used; a
  lever the state sets; a weakening gain against the closed form; the reference plant pinned to
  1e-9; the units of the instrument, the levers and the state at 1e-6 and 1e6; a weight that drops
  every row the instrument moves.

## Alternatives

- **Raise on an instrument short of rank.** Rejected: a fit with nothing that identifies it
  already returns `identified=False` with the observational fit, and every caller reads that field
  first.
- **Fall back on `adjust_for` where it is given.** Rejected: the instrument says the covariates
  leave a confounder.
- **The flag kept where the log moves no direction**, as the adjusted path keeps it. Rejected:
  an instrument of zeros on a log whose lever the state sets would read identified with a rank of
  0, the defect this decision removes.
- **The rank on every coefficient, `unmoved`'s directions included**, the textbook full column
  rank. Rejected: a lever the log never used would leave every instrument unidentified, where the
  adjusted fit of the same log is identified with that lever's directions in `unmoved`, so the flag
  would mean two things on one log.
- **The rank of the moment the fit solves**, `Z'D`, or of `Z'D + ridge I`. Rejected: the ridge
  makes the second full rank always. In the first, `Z` keeps the part the covariates predict, and
  the action residuals it meets are cross-fitted, and orthogonal to the nuisance's features but not
  to their products with the channel's. So it reads noise around zero, not zero: with the plant's
  instrument set to zeros, its singular values were 1.9e-4 to 3.2e-3 a row, full rank, where the
  plant's instrument gave 0.42 to 0.74.
- **A threshold on a first-stage F or on Cragg and Donald's statistic**, against Stock and Yogo's
  critical values. Rejected for this change: a threshold is a test of weakness with a size, which is
  the inference this does not do. The correlations are what such a statistic is made of.
- **Singular values of the moment scaled to the largest.** Rejected: they move with the scale of
  the channel's features against one another; the canonical correlations move with neither side's
  units.
