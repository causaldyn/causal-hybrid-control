# ADR 0068 — A panel holds values that cannot change in place

**Status:** accepted, 2026-10-07.

## Context

ADR 0056 made a panel hold a read-only copy of each column, so that its data cannot change under
the hash it records. For an object column, that copy holds the caller's objects. A write to a dict
the caller still held changed the panel: after `meta[1]["tag"] = 999`, `panel["meta"][1]` read
`{'tag': 999}`, and `data_sha256` no longer named the panel's data. The read-only flag guards the
cells, not the objects in them, so a write through `panel["meta"][1]` did the same. A list as a
unit failed the index check with a bare `TypeError`.

The finite check read the columns of a floating dtype only. An object column of floats passed nan
and the infinities: `Panel.wide` returned them, and `chc.decision` read them as float64 among its
states, to fail later with a message that named no row. A `Decimal`'s nan, NumPy's `float32` among
objects, and a complex column passed the same way.

A review of 0.14.2 reproduced both.

## Decision

- **An object column holds values of the types below, or of their subclasses.** `str`, `bytes`,
  `bool`, `int`, `float`, `complex`, `Decimal`, `Fraction`, `UUID`, `date`, `time`, `datetime`
  and `timedelta`, and NumPy's scalars of these kinds (`np.bool_`, `np.number`, which holds
  `np.timedelta64`, and `np.datetime64`; `np.str_` and `np.bytes_` subclass `str` and `bytes`).
  None of them changes in place, and the text of each names its value. Each value is hashed as
  ADR 0056 hashes it: by its type and its text.
- **Any other value is refused, not frozen.** `PanelError` names the column, the value, its type,
  and the unit and the time of row 0, which holds the column's one type. This covers what changes
  in place (a dict, a list, a set, a bytearray, an array, an object of the caller's own class),
  the missing values a frame hands over (`None`, pandas' `NA`), and the types the library cannot
  vouch for (a tuple, a pandas `Period`). A value whose text is its address in memory keeps the
  reason ADR 0056 gave.
- **A number that has a nan or an infinity is checked as a float column is.** In an object column
  that is a `float`, a `complex` or a `Decimal`, NumPy's inexact scalars among them, and a complex
  column is checked too. A nan or an infinity is refused with the message a float column gives,
  which names the column, the unit and the time. A `Decimal` answers by its own `is_finite`, since
  `float` raises on its signalling nan. An `int`, a `Fraction` and a `bool` have no nan. A NaT of
  `datetime64` or `timedelta64` is held, as a datetime or a timedelta column holds one.
- **The values are checked before the index.** A unit or a period that is not hashable, a list or
  a `Decimal`'s signalling nan, is refused with `PanelError`, as every other value is.

## Consequences

- Every panel that is still accepted hashes as before. A test pins the digest that 0.14.2 gave a
  panel with a column of each type held.
- A `bytes` value or a NumPy string whose text reads as an address, `b'<object at 0x10>'` say,
  is now held. ADR 0056's address check reads only the types refused, since the text of a type
  held names its value.
- A panel that held any other value is refused: a column of dicts, of lists, of tuples, of pandas
  `Period`s, or of `None` in every row. Convert the column first: a dict's fields to columns of
  their own, a tuple to text, a `Period` to a timestamp (`.dt.to_timestamp()`) or to text.
- A missing number is refused in each form a frame hands it over: nan in a float, a complex or an
  object column, `None`, and pandas' `NA`.
- *Left*: a subclass whose text reads state of its own, a `datetime` or a `time` whose `tzinfo` is
  an object of the caller's own class, and a value changed through its private state (a
  `Fraction`'s slots, or `object.__setattr__`), can still change under the hash. A column
  that is NaT in every row is held: pandas' NaT is a `datetime`, NumPy's is a `datetime64`, and a
  datetime column holds a NaT too. A number that is finite in its own type but past float64's
  range, a `Decimal` of `1E+400` or a `longdouble` say, is held. A reader that takes float64 read
  it as an infinity, as it did from a `longdouble` column; it now refuses it (ADR 0072).

## Alternatives considered

- **A deep copy at construction.** Rejected: the caller's objects no longer reach the panel, but
  the panel's own objects still change through `panel[name][row]`. Also, an arbitrary object need
  not have a deep copy.
- **Freezing: a dict to a mapping proxy, a list to a tuple, a set to a frozenset.** Rejected: it
  changes the values' types, and so their hash and what a reader gets back. A mapping proxy does
  not pickle, and no rule freezes an object of the caller's own class. The text of a container is
  its items' `repr`, which NumPy 2 changed for its scalars. The order of a set's items follows the
  string hash, which changes from one process to the next.
- **Any hashable value.** Rejected: Python promises that a hashable value's hash does not change,
  not that the value does not change. An object of the caller's own class hashes by its identity,
  and a write to it still changes its text.
- **Exact types, with no subclasses.** Rejected: pandas hands over a column of time-zone-aware
  timestamps as `Timestamp` objects, which subclass `datetime`.
- **Pandas' `Period` among the types.** Rejected: pandas is not a dependency, so the panel could
  know the type only by importing pandas or by its name, and a `Period` column converts in one
  call.
- **Numbers refused in an object column.** Rejected: ADR 0056 holds them, a column of `Decimal`s
  is an object column, and a panel of finite numbers that was accepted would be refused.
- **Each number checked as float64, the precision the readers take.** Rejected: it would refuse a
  finite value as not finite, where a `longdouble` column is checked in its own precision, which
  this decision leaves as it was. `float` also raises on a `Decimal`'s signalling nan.
