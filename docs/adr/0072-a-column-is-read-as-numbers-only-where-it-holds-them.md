# ADR 0072 — A column is read as numbers only where it holds them

**Status:** accepted, 2026-10-07. Amended 2026-10-07: the entry points that take a caller's
data as arrays read it by the same rule, and the logger check reads what the rule reads.

## Context

The readers of a caller's columns cast them with `np.asarray(column, dtype=float)`, whatever their
dtype. NumPy's cast to float64 reads far more than numbers:

| the column holds | the cast reads |
|---|---|
| the text `"1.5"`, `"nan"`: an object, a fixed-width or a NumPy string column | 1.5, nan |
| dates, `datetime64[D]` | days since 1970; NaT as -9.2e18 |
| durations, `timedelta64[s]` | seconds |
| complex numbers | the real part, with a `ComplexWarning` |
| a `Decimal` of `1E+400`, a `longdouble` of 1e400 | an infinity |
| an `int` or a `Fraction` of 400 digits | a bare `OverflowError` |
| a NumPy masked array | the values under its mask |

A panel holds text, dates and times: a unit's name, a period's date (ADR 0056, 0068). ADR 0068
held a number finite in its own type but past float64's range, and left the reader to take it as
an infinity. So the readers took whatever the panel held: `Panel.wide`, and `prescribe`'s fit,
logger check, start and driver range, `Prescription.evaluate`'s windows, and the marketing-mix
case study's start. `prescribe` made a plan from a lever logged as text, the plan the numbers it
spells give, and from one logged as dates. pandas 3 hands its `str` dtype over as text among
objects, and polars its text as a fixed-width string, so either reached the readers unchanged.

`chc.gmethods` cast every column of the caller's frame, those it never read among them: a text
treatment read as numbers, and a text label column failed the call. `as_columns` handed a masked
array on, or converted it with `np.asarray`, and the mask was lost either way.

The entry points that take a caller's data as arrays, or as a mapping of arrays, cast it the same
way: NumPy's cast as the table says, JAX's by refusing text, dates and objects with a message that
named no argument and by taking complex numbers into the fit. Of 207 such arguments of 85 entry
points, 144 read one of the values above without an error. `chc.estimators` cast every column of
the frame, as `chc.gmethods` did, and `lalonde_ate` standardised a float32 caller's covariates in
float32. The logger check read a column only of a NumPy number dtype, so it took a column of
booleans, which a panel reads as 0 and 1, for one it cannot read.

## Decision

- **One rule says what is read as numbers.** A column whose dtype is boolean, integer or floating,
  and an object column of `bool`, `int`, `float`, `Decimal` or `Fraction` values or NumPy's scalars
  of these kinds. Not read: text (dtype kinds `U`, `S` and NumPy's `StringDType`), dates and times
  (`M`), durations (`m`), complex numbers (`c`), and in an object column any other value, a
  `datetime`, a `UUID`, `None`, pandas' `NA` and NumPy's `timedelta64` among them. Nor is a number
  finite in its own type that is an infinity in float64. It lives in `chc.frames`, and returns the
  first value refused and why; each reader phrases the refusal.
- **A panel holds such a column and refuses to read it as numbers.** The panel cannot know at
  construction which columns a reader will take as numbers, and its labels are text and dates.
  One method reads a panel's column as float64 for every reader: `Panel.wide`, `prescribe`'s fit,
  logger check, start and driver range, `Prescription.evaluate`'s windows and the case study's
  start. It raises `PanelError` naming the column, the value, its unit and its time: the first
  value refused in an object column or a wider float, row 0 where the dtype is refused.
- **A reader of a caller's frame reads only the columns it names.** `sequential_g_formula` and
  `naive_pooled_effect` read the outcome, the treatments and the confounders through the same
  rule, and raise `ValueError` naming the column, the value and the row. A column no argument
  names is not read.
- **A masked cell is refused where a column enters.** `as_columns`, the door every named column
  passes, a panel's among them, raises `ValueError` naming the column, the row and how many cells
  are masked. No reader of named columns takes a missing value, and the values under a mask are
  none of the column's. A masked array that masks no cell is its data.
- **A nan or an infinity is not this rule's.** A panel refuses one at construction, as before
  (ADR 0068). A g-method reads it, and its effect reads nan.
- **A dtype NumPy holds as neither numbers nor text is left to the cast.** The cast refuses a
  structured dtype, and reads `ml_dtypes`' `bfloat16`, of kind `V`, as the number it is.
- **An entry point reads a caller's arrays through the same rule as they enter.** One reader in
  `chc.frames` takes an argument and its name, refuses what the rule does not read, naming the
  argument, the value and where it lies, and hands the argument on as the caller gave it; the
  entry point casts it after, as before. A JAX array holds no text, dates or objects, and under
  `jax.jit` its values are unknown, so its dtype alone decides, and a traced entry point keeps
  tracing. A mapping's entries and a frame's columns are read only where an argument names them.
  The rule applies to what a caller logged: logs, samples, series, outcomes, treatments,
  covariates, histories, stored decisions, forecasts and a target's levels, `prescribe`'s start
  and a policy's state. `Target`, `Driver` and `prescribe`'s start raise `DecisionError`, the
  others `ValueError`.
- **`lalonde_ate` reads its data in float64**, as a panel reads its columns, whatever the
  caller's dtype; the estimator casts it to its own after.
- **The logger check reads what a panel reads as numbers.** A column of a boolean, an integer, a
  floating, a complex or a duration dtype, or an object column the rule reads; the panel's reader
  refuses the complex and the durations among them.

## Consequences

- A column the rule reads is read as before, bit for bit: the output is the same cast of the same
  column. Tests pin each width of boolean, integer and float, pandas' nullable numbers with no
  value missing, and object columns of each type read.
- A panel is built and hashed as before: the digests ADR 0056 and 0068 pinned are unchanged.
- A panel of a lever, a state, a covariate or a driver that is not numbers fails `prescribe` with
  `PanelError` before the fit; before, it was fitted.
- The check costs a dtype test on a column of numbers, a pass over a float wider than float64,
  and a pass over an object column, at each read.
- The logger check reads the columns of a numeric dtype it read before, complex numbers and
  durations among them; such a column it reads is now refused, not read as its real part or its
  count of seconds.
- A g-method no longer fails on a text column it does not read, and an estimator in
  `chc.estimators` names a masked column where JAX refused it without the name.
- An entry point reads numbers as before, bit for bit: it receives the caller's own array and casts
  it as it did. A masked array that masks no cell is now its data where JAX refused it, and a
  float32 caller's LaLonde data reads as its float64 widening. A stored decision's field that holds
  a NumPy array of objects with no dimensions, which NumPy keeps whole among a list's values, is
  refused, as a panel refuses one; it was read as its value.
- An estimator in `chc.estimators` no longer fails on a column it does not read, and its EconML and
  DoWhy adapters refuse a column that is not numbers before they import those packages.
- The logger check runs where a lever's parent is a column of booleans, and tests a column of
  booleans beside the plan; it skipped both.
- The reader costs a dtype test on an array of numbers and a pass over an object array, and a list
  is converted by `np.asarray` once more.
- *Left*: an argument that is a model's rather than a log's is cast as before: a box's bounds in
  `chc.allocation`, a Toeplitz operator's columns and generators and `levinson_durbin`'s
  autocorrelation, a channel's response at a spend, `cvar_upper`'s values, and the states and
  actions a fitted model is asked about (`predict`, `rollout`, an estimate's `cate`, the planners'
  start). So are labels --- clusters, units, periods, strata --- which are read as labels, not as
  numbers. An object array of numbers reaches JAX as NumPy holds it, and JAX refuses it, as before.

## Alternatives considered

- **Refuse at construction every column that is not numbers.** Rejected: a panel's units, periods
  and labels are text and dates, which ADR 0056 holds, and most panels with named units would be
  refused.
- **Refuse at construction only a number past float64's range.** Rejected: ADR 0068 holds a number
  finite in its own type, and a column no reader takes as numbers, an identifier of `Decimal`s say,
  need not fit float64.
- **Record each column's verdict at construction, for the readers to consult.** Rejected: a field
  on a frozen dataclass and in its pickle, for a check that costs a dtype test on a column of
  numbers. The read is where the question is asked.
- **A guard in each reader.** Rejected: six readers, each a copy free to drift from the others.
- **Read a masked cell as nan.** Rejected: a panel refuses nan, so the cell would be refused with a
  message about a nan the caller never wrote; a g-method's effect would read nan; and an integer
  column would turn float.
- **Parse text that spells a number.** Rejected: `"nan"`, `"inf"` and `"1e400"` spell numbers too,
  and a column of text is the caller's to convert.
- **`numbers.Real` as the test of a number.** Rejected: neither `Decimal` nor NumPy's `bool_` is
  registered as one.
- **A reader of arrays that returns them cast.** Rejected: the entry points cast to different
  precisions on purpose --- float64 in NumPy, the x64 flag's in JAX, a float32 caller's own where an
  estimator keeps it --- and NumPy's cast of a traced array fails. Handed on uncast, each entry
  point's numbers are what they were.
- **A check of its own at each entry point.** Rejected for the reason a guard in each reader was:
  some 85 copies, each free to drift from the rule.
