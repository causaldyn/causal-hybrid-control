# ADR 0056 — A panel hashes an object column by its values

**Status:** accepted, 2026-10-06.

## Context

`Provenance.data_sha256` is meant to name a dataset, so that a number can be reproduced from it.
`Panel.from_frame` hashed each column's name, dtype, shape and `tobytes()`. Two things broke that:

- **An object column's bytes are its pointers.** A unit column of strings read from pandas is an
  object column. Two panels of the same strings hashed apart, even within one process, and no
  other run could rebuild the hash.
- **The panel held views of the caller's arrays.** A write to the source array changed the
  panel, and its recorded hash stayed as it was.

A review of 0.13.0 reproduced both.

## Decision

- **An object column is hashed by its values.** For each value, the hash reads its type's
  `__qualname__` and then its `str`, each as UTF-8 (`surrogatepass`) behind an 8-byte length.
  The lengths keep values from running together, and the type keeps `1` apart from `"1"`. Any
  other dtype is hashed by its bytes, as before.
- **An object column holds values of one type.** A second type is refused with `PanelError`,
  naming the column, the value, the unit and the time. In practice it is a missing value among
  strings: pandas reads one as nan, polars as None. A float column's nan is refused the same way.
- **An object column whose values' text is their address in memory is refused.** That covers a
  text containing `" at 0x"`, as CPython's default `repr` writes for a plain object, a function
  or a method. Such text names no value another run could rebuild.
- **The panel owns its data.** Each column is copied before it is checked, so the bytes checked,
  hashed and held are the same bytes. The copies are read-only. They sit in `_Columns`, a mapping
  that takes no new column. Pickling and deep copies rebuild it through its constructor, which
  sets the flag again.

## Consequences

- A panel with no object column hashes as before; a test pins one such digest. A panel with an
  object column hashes otherwise, in releases whose decision record has `schema_version` 2.
- The same strings hash the same whether they come from a mapping or a pandas frame. Polars reads
  strings as fixed-width unicode (`<U…`), which is hashed by its bytes. So a polars panel and a
  pandas panel of the same strings still hash apart, by dtype, as two dtypes always have.
- Building a panel copies the data once.
- An in-place write to a panel column raises `ValueError`. Write to a copy instead.
- *Left*: a type whose `str` is lossy hashes lossily. For example, a class whose text does not
  name its fields gives two different values one hash.

## Alternatives considered

- **`pickle.dumps` of each value.** Rejected: the bytes depend on the pickle protocol and on the
  library version that defines the type.
- **`repr` instead of `str`.** Rejected: NumPy 2 changed a scalar's `repr` (`np.float64(0.1)`),
  but not its `str`.
- **`astype(str)` and hashing the fixed-width bytes.** Rejected: it drops the type, and pads to
  the widest value.
- **`types.MappingProxyType` for the columns.** Rejected: it does not pickle, so a panel could not
  reach a worker process.
- **Refusing object columns outright.** Rejected: pandas reads strings as object columns, so most
  panels with labelled units would be refused.
