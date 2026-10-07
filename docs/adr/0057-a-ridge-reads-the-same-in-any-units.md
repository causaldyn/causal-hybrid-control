# ADR 0057 — A ridge reads the same in any units

**Status:** accepted, 2026-10-07.

## Context

`fit_causal_residual` stabilises four solves with `ridge`, 1e-6 by default: the nuisances, the
channel's moment, the drift regression and the instrument's first stage. The nuisances read
standardised covariates, so their ridge sits on columns of unit size. The other three added
`ridge` times the identity to a Gram matrix of the caller's raw columns: the moment's of the
actions times the state's features, the drift regression's of the state's monomials and the
drivers, the first stage's of the monomials of the state and the instrument. A column logged in
units `s` times its own enters that Gram at `s^2`, so one `ridge` weighed it `s^-2` times as much.

At unit scale the ridge is about a billionth of the Gram's diagonal. In small units it took over,
and nothing was raised. On the tests' log:

- with the actions in millionths of their units, the channel read 0.0008 and -0.0004 where the
  fit reads 0.805 and -0.398;
- with the state in millionths, the drift's slope read -0.0020 where it reads -0.475;
- with a driver in millionths, its gain read 0.0013 where 1.198, and the channel 0.778 where
  0.796;
- with the instrument in millionths, the channel moved by up to 0.36;
- `solve_channel_moment`, on residuals in millionths, read 0.0016 where 0.80.

In large units the ridge only vanished, so the defect showed in one direction alone. A share, a
conversion rate or spend in millions logged in units are all small columns.

A unit has an origin as well as a size: a temperature in kelvin is one in degrees Celsius moved by
273.15. A ridge that penalises the bias, or that scales a column's term by its mean square, which
grows with the column's distance from zero, shrinks a slope by about `ridge (c/s)^2 / n` for a
column `c` from zero with spread `s` over `n` rows. With the state ten thousand spreads from zero,
the drift's slopes moved by 0.036 of the largest, and a ridge solve at 0.1 on 50 rows kept 0.0006
of the slope of a column a thousand spreads from zero.

Two floors of the fit were in the caller's units too:

- The nuisances standardise each column, but took one whose spread was under 1e-12 for a constant
  and left it unscaled under their ridge. The confounder logged at 1e-13 of its units was not
  adjusted for, and the channel moved by 1.27; the state at 1e-13 moved it by 0.46.
- Where the moment has no data, the fit reads the drift's response by least squares on the raw
  drift design (ADR 0054). Its rank cutoff is relative to the largest singular value: with the
  state at 1e-12 of its units the state's columns fell under it beside the constant's, and the
  channel moved by 50; at 1e15 the constant's did, and it moved by 0.02.

## Decision

**Each ridge term is scaled by its column's second moment about what the fit absorbs for free.**

- **A design with a bias** leaves the bias free, and scales every other column's term by its
  variance about its mean. The bias takes up a column's mean, so the term reads the column's
  spread, and the solve reads the same in any units and any origin of a column. That covers the
  drift regression, its covariance, the instrument's first stage and the drift's response to an
  unmoved direction: each design's first column is its bias.
  - A column constant to within 1e-12 of its own root mean square keeps its mean square. It is
    then penalised beside the bias it duplicates, and the Gram stays invertible.
  - Such a design is solved on its centred columns: the slopes from their Gram and the penalty,
    the bias from the columns' and the target's means, and the inverse that the standard errors
    and the influences read is mapped back to the raw columns. The raw Gram with the same penalty
    is the same solve in exact arithmetic, but its rounding grows with each column's mean square,
    not its variance.
- **A design with no bias** scales each column's term by its mean square, the Gram's own diagonal
  over the rows. Each column of the channel's moment is an action times a feature of the state, so
  none is free.
  - The moment's penalty is the mean square of each column of the channel's design on the log's
    raw actions, its rows weighted as the moment weighs them. That is the size ADR 0054's split
    scales the unmoved directions to, and the size the ridge meant at unit scale. Where the moment
    is solved on the moved directions alone, the penalty is taken onto them.
  - `solve_channel_moment` is given residuals, not actions, so its penalty is its regressor's own
    mean square, weighted the same way.
- A column of zeros keeps a term of 1: its coefficient is zero either way.
- A column is constant to the nuisances when its spread is within 1e-12 of its own root mean
  square, not of 1.
- The least squares on the drift's design scales each column by the power of two nearest the
  reciprocal of its root mean square first. The cutoff then falls at one share of every column,
  and the rescaling is exact, as in the independence tests.

## Consequences

- Every fit moves, by the ridge's share of each term. On the tests' logs at unit scale, a
  coefficient moved by at most 6e-9 of the largest, a standard error by 1.3e-8 and a plan's
  actions by 4e-10. A default's numbers move, so the change waits for a minor release.
- The channel times the actions' units, the drift and the channel with the state in other units,
  each row's influence, the channel with the adjustment set in other units, a driver's gain times
  its units and the channel under an instrument in other units read the same from 1e-15 (1e-13
  for the state) to 1e15, to 1e-6 and in most cases to 1e-14. Tests hold each.
- With the state a hundred or ten thousand spreads from zero, the drift's slopes and the channel
  read the same to 5e-15 and 4e-13 of the largest slope, the rounding of the state's own values at
  that distance, and the intercept moves by the slopes times the offset. A test holds each. The
  channel's own terms are scaled by their mean squares, so the shift is absorbed exactly where the
  channel does not vary with the state, and approximately where it does: a shift moves an action
  times the state into an action times the bias.
- The ridge still shrinks a direction the log moved little next to the actions' own size, as it
  did at unit scale. That is what it is for: ADR 0054 takes the directions the log never moved
  out of the moment, and the ridge steadies the ones it barely moved.

## Alternatives considered

- **Scale every column to unit size, solve, and scale the coefficients back.** The same solve,
  written in other units; the penalty scaled by each column's second moment writes it in the
  caller's.
- **Standardise the columns, as the nuisances do, and solve in those units.** The same solve: the
  centred solve already maps the bias back, and the penalty does the scaling, so the coefficients
  stay the model's own with no step to undo.
- **The raw Gram with the centred penalty.** The same solve in exact arithmetic. Rejected: its
  rounding is about the machine epsilon times the rows times each column's mean square, so a
  column whose spread was 1e-11 of its size read nan.
- **The mean square for every column, the bias penalised with the rest.** Rejected: it reads the
  same in any size of a column's units but not in any origin, and it shrank the slope of a column
  far from zero by the square of its offset over its spread, as the context measures.
- **One ridge relative to the Gram's trace.** Rejected: columns in different units would still
  weigh it differently.
- **The moment's penalty from the residuals' size in the fit.** Rejected: an action the covariates
  nearly determine leaves small residuals, so the ridge would let go of the directions it exists
  to steady. ADR 0054 scales by the raw actions for the same reason.
- **Refusing columns in small units.** Rejected: the units are the caller's, and no threshold
  tells small from wrong.
