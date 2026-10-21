# ADR 0126 — The media mix's opt-in readings are the defaults

**Status:** accepted, 2026-10-21. Amends ADR 0033, ADR 0109 and ADR 0114.

## Context

Three readings of the media-mix layer shipped opt-in, each with a `FutureWarning` on the old
default that said a later release would change it:

- `fit_lift` read each test's noise as independent over its periods unless given `serial="ar1"`.
  A geo gap's noise persists from week to week. Read as independent, AR(0.8) noise made the
  check reject the true channel on 29.5 % of worlds at a stated 5 %.
- `roi` and `marginal_roi` cut the carryover at the series' end unless given `tail=True`. Through
  a kernel of retention 0.9 and length 30, the last of 52 periods read 0.0003 cut and 0.0165 with
  its whole tail.
- `pessimistic_equilibrium_allocation` planned through the simulator's own equilibrium unless
  given `equilibrium=fit_equilibrium(logs)`. No log supplies that equilibrium, and through it a
  response of zero planned as close to the oracle as the de-confounded one.

The library ships a behaviour change opt-in first, with a warning, and makes it the default one
release later.

## Decision

- `fit_lift(serial=)` defaults to `"ar1"`. `serial="independent"` keeps the independent reading
  and its Breusch–Godfrey check of the premise, with the WARNING that names `"ar1"`.
- `roi(tail=)` and `marginal_roi(tail=)` default to `True`, all of a `Channel`'s tail.
  `tail=False` keeps the cut, and a whole number extends the series by that many periods.
- `pessimistic_equilibrium_allocation` requires `equilibrium=`. Without it the planner raises
  `TypeError` with a message that names `fit_equilibrium`. It no longer reads the market's base
  demand, attractiveness or solver.
- The three `FutureWarning`s are removed.

## Consequences

The lift table, re-measured under the new default on 200 worlds per persistence of four go-dark
tests of 16 periods (Monte Carlo SE 1.5 % at 5 % and at 95 %):

| AR(1) coefficient | default (`"ar1"`): check rejects / coefficient covers | `"independent"` | Breusch–Godfrey rejects |
|---|---|---|---|
| 0 | 7.0 % / 95.0 % | 4.0 % / 96.5 % | 7.0 % |
| 0.3 | 5.0 % / 96.5 % | 11.5 % / 94.0 % | 54.0 % |
| 0.5 | 7.0 % / 98.0 % | 16.0 % / 91.5 % | 93.5 % |
| 0.8 | 3.5 % / 97.0 % | 29.5 % / 86.5 % | 100 % |

- The default holds its level at every persistence. On independent noise it rejects 7.0 %, 1.3
  Monte Carlo SE above 5 %, and spends a degree of freedom. The retention's interval covers 92.5 %
  there and the scale's 93.5 %.
- A lift fit with no more periods than its parameters, a noise variance and a coefficient is now
  refused by default. It was fitted before with one degree of freedom more.
- `roi` and `marginal_roi` read the same as before on a window whose carryover ends inside the
  series. Near the series' end they now read the whole tail. On a function of the series that is
  not a `Channel` the default raises `TypeError`, since its kernel's length is not known: the
  caller passes the periods, or `tail=False`.
- A caller of the marketplace planner that passed no equilibrium gets a `TypeError`.

## Alternatives

- **Keep the old defaults and the warnings for another release.** Rejected: each warning said
  the change was coming, and the old readings are wrong in typical use.
- **Make `equilibrium` a required keyword with no default.** Rejected: Python's own error names
  the argument, not `fit_equilibrium`, and the message is where the caller learns what to pass.
- **Cut the carryover silently for a function that is not a `Channel`.** Rejected: that is the
  silent cut the default removes. The error says what to pass.
