# ADR 0058 — A panel's calendar is declared

**Status:** accepted, 2026-10-07. Amended 2026-10-08: from 0.16.0, undeclared dates off a uniform
grid are refused, as 0.15.0 announced.

## Context

`prescribe` fits a transition on two rows of a unit whose periods are consecutive. Which two
periods are consecutive was inferred from the stamps:

- **Numbers and dates on a uniform grid** were placed on the grid of their least spacing. A period
  no unit logged then still parts the two on either side of it.
- **Anything else was ranked:** the periods logged were the grid. Months, quarters and years
  stamped as dates are on no uniform grid, being 28 to 31, 90 to 92 and 365 or 366 days apart.

A review of 0.13.0 found the consequence. In a monthly log with July missing for every unit,
June and August were ranked as consecutive, and a transition was fitted across a month nobody
logged. No spacing rule can repair this, because the stamps do not name their calendar: month
ends, stamps four weeks apart and an irregular sample can show the same spacings.

## Decision

- **`Panel.from_frame` takes `frequency`.**
  - For a date column: `"D"`, `"W"`, `"M"`, `"Q"` or `"Y"`. A row's period is the day, week,
    month, quarter or year its stamp falls in, by numpy's calendar units at any resolution.
    Numpy's weeks run from a Thursday, as 1970-01-01 did. A quarter is three months from January.
  - For a numeric column: a positive step. A period is a whole number of steps from the first
    period. Whole numbers over a whole step are counted exactly; other numbers to a part in 1e9.
    A period off the step is refused, naming its row.
  - `"observed"`, for any column: the periods logged, in their order.
- **The periods are the declared ones.** A transition joins two rows of a unit one period apart.
  A unit with two rows in one period is refused, naming both rows and their stamps.
- **The rows of one period share one label: the earliest stamp logged in it.** In the common
  case, one stamp per period, the labels are the caller's own stamps, as they were before.
  `periods`, `codes`, `wide` and the balance check group the rows by period.
- **Undeclared, the reading is unchanged**, and dates off a uniform grid emit a `FutureWarning`
  that names `frequency`. From 0.16 they are refused unless a frequency is declared.
  - *From 0.16.0:* they are refused with a `PanelError` that names `frequency`;
    `frequency="observed"` gives 0.15.0's reading.
- **The frequency is part of the dataset's identity.**
  - `Provenance` records it, and its JSON writes it.
  - The hash takes it in where one is declared, so an undeclared panel hashes as before.
  - A step is recorded as a float, so that `7` and `7.0` are one reading.
- **A `NaT` is refused, as a nan is.** A date that is missing has no calendar reading.

## Consequences

- One monthly calendar stamped as months, as month ends or in nanoseconds gives the same
  transitions. A year's end, and a February stamped on the 29th, are one step like any other.
- `wide` still has a column for each period logged. A period that no unit logged has no column,
  and a DiD or a synthetic control on that matrix reads its columns as consecutive. That is
  unchanged by this decision.
- A weekly log stamped on mixed weekdays can split one business week in two, because numpy's week
  starts on a Thursday. Stamps on one weekday are consecutive under any anchor.
- *Left*:
  - Text periods such as `"2024-01"` cannot declare a calendar. Convert them to dates.
  - Numbers off a uniform grid with no `frequency` are still ranked, and without a warning.

## Alternatives considered

- **Infer months from spacings of 28 to 31 days.** Rejected: inference is what failed, and the
  same spacings fit a four-weekly log.
- **pandas periods and offsets.** Rejected: pandas is not a dependency of the library.
- **Label a period by its start, such as 2024-01-01 for January.** Rejected: it renames the
  periods of every panel with one stamp per period, the common case.
- **Refuse undeclared dates off a grid at once.** Rejected: it breaks every monthly panel
  without notice. One release warns first.
- **ISO weeks, starting on Monday.** Rejected: numpy's units need no calendar code of their own,
  and consecutive stamps on one weekday are consecutive under either anchor.
