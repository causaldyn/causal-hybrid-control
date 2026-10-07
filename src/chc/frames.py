"""Accept a columnar frame -- pandas, polars, or a plain mapping -- wherever chc takes named data.

Neither pandas nor polars is a chc dependency, so a frame is recognised *structurally* rather than
by import: both expose ``.columns`` and ``frame[name]``, and every column converts through
``np.asarray``. Estimators call :func:`as_columns` once on entry and index a plain dict afterwards.

The mapping branch is guarded by an ``isinstance`` rather than by duck-typing, because a frame that
is not a ``Mapping`` still survives the mapping idioms with the wrong answer instead of an error:
``dict(polars_frame)`` and ``for name in polars_frame`` both iterate columns *as values*, handing a
caller column data where it asked for column names.

A column read as numbers holds real numbers. NumPy's cast to float64 takes far more: it reads the
text ``"1.5"`` as 1.5, a date as its count of days since 1970, and a ``Decimal`` of ``1E+400`` as
an infinity. So a reader of numbers reads a caller's column through one check, which refuses
text, dates, times, durations, complex numbers and any other object that is not a real number,
and a real number that is finite but past float64's range, naming the column and the row. An entry
point that takes a caller's data as arrays, not as named columns, reads each through the same
check as it enters (:func:`_real_numbers`), naming the argument.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any, Protocol, TypeVar

import jax
import numpy as np
from numpy.typing import NDArray

# The values a reader of numbers takes from an object column. ``int`` covers ``bool``; NumPy's
# integers cover ``timedelta64``, a duration, which ``_not_numbers`` refuses by name.
_REAL: tuple[type, ...] = (int, float, Decimal, Fraction, np.bool_, np.integer, np.floating)
# The dtypes whose values are no real numbers, by kind: text, of fixed width or not, dates and
# times, durations, and complex numbers. A float64 cast reads the first three as numbers and the
# last as its real part, with a warning at most.
_NOT_REAL = {
    "U": "text",
    "S": "text",
    "T": "text",
    "M": "dates or times",
    "m": "durations",
    "c": "complex numbers",
}
_NOT_READ = "so the column is not read as numbers"
_PAST_FLOAT64 = "a finite number past float64's range, where it is an infinity"

_Value = TypeVar("_Value")


class ColumnFrame(Protocol):
    """A frame with named columns, indexable by name: pandas, polars, pyarrow, and friends."""

    @property
    def columns(self) -> Sequence[str]: ...

    def __getitem__(self, name: str, /) -> Any: ...


ColumnData = Mapping[str, Any] | ColumnFrame
"""What every chc entry point that takes named data accepts."""


def as_columns(data: ColumnData) -> dict[str, Any]:
    """Normalise ``data`` to ``{name: column}`` from a mapping or any columnar frame.

    The values are ``Any`` rather than ``ArrayLike`` and that is the honest type, not a shortcut: a
    mapping's columns come back exactly as they went in, so their type is the caller's, and every
    consumer here re-asserts it with the `asarray` its own precision contract requires.

    A mapping passes through **unconverted**, which is what keeps this cheap enough to sit on every
    entry point: the library's own callers hand over JAX arrays, and `np.asarray` on those would
    force a device-to-host copy and raise outright on a tracer inside `jax.jit`. Only the frame
    branch materialises, and it stops at NumPy -- precision is the caller's decision, and the two
    consumers disagree on purpose (:mod:`chc.gmethods` is float64 by contract, the estimators follow
    JAX's x64 flag).

    A NumPy masked array is read through its mask, in either branch. A masked cell is a missing
    value, and no reader of named columns here takes one --- a panel refuses a nan, and a fit would
    read what lies under the mask as data --- so a column with one is refused. A masked array
    that masks no cell is its data.

    Raises:
        TypeError: when ``data`` is neither a mapping nor a frame with ``.columns``.
        ValueError: when two columns go by one name as text -- ``0`` and ``"0"``, or a label a
            pandas frame repeats -- so that one would replace the other; or when a column is a
            masked array that masks a cell, naming the column, the row and how many are masked.
    """
    if isinstance(data, Mapping):
        return _by_name(
            (label, _unmasked(f"column {str(label)!r}", column)) for label, column in data.items()
        )
    names = getattr(data, "columns", None)
    if names is None:
        msg = (
            "expected a mapping of column name -> array, or a frame exposing `.columns` "
            f"(pandas / polars), got {type(data).__name__}"
        )
        raise TypeError(msg)
    return _by_name(
        (name, np.asarray(_unmasked(f"column {str(name)!r}", data[name]))) for name in names
    )


def _unmasked(name: str, column: Any, error: type[ValueError] = ValueError) -> Any:
    """``column`` as it is, or a masked array's data where it masks no cell; ``np.asarray`` drops a
    mask and keeps what lies under it. ``name`` is how the refusal names the column."""
    if not isinstance(column, np.ma.MaskedArray):
        return column
    masked = np.flatnonzero(np.ma.getmaskarray(column))
    if masked.size:
        raise error(
            f"{name} is masked{_where(column.shape, int(masked[0]))} "
            f"({masked.size} of {column.size} cells): a masked cell is a missing value, which chc "
            "does not read; fill it, or drop its row"
        )
    return np.ma.getdata(column)


def _numbers(column: Any, name: str) -> NDArray[np.float64]:
    """A caller's column as float64, as NumPy casts it, where it holds real numbers; refused with
    ``ValueError`` naming the column and the row where :func:`_not_numbers` finds it does not."""
    array = np.asarray(column)
    problem = _not_numbers(array)
    if problem is not None:
        position, why = problem
        raise ValueError(
            f"column {name!r} is {array.reshape(-1)[position]!r} at "
            f"{_at(array.shape, position)}: {why}, {_NOT_READ}"
        )
    return np.asarray(array, dtype=np.float64)


def _real_numbers(value: _Value, name: str, error: type[ValueError] = ValueError) -> _Value:
    """``value``, a caller's data, as the caller gave it, where it holds real numbers by
    :func:`_not_numbers`; a masked array that masks no cell as its data.

    The rule a panel's columns are read by, for an argument: each entry point that takes a caller's
    data as an array passes it through here as it enters, and casts it after as its own precision
    requires. Nothing is converted, so data that is numbers is read as before, bit for bit. A JAX
    array holds no text, dates or objects, and is checked by its dtype alone: under ``jax.jit`` its
    values are not known, and a traced entry point keeps tracing. A list or another sequence is
    read as ``np.asarray`` reads it, so one that holds a traced JAX array is refused by NumPy.

    Raises:
        ValueError: or ``error``, naming ``name``, the value refused and where it lies, or the
            masked cells and how many.
    """
    if isinstance(value, jax.Array):
        problem = _not_numbers(value)
        if problem is not None:
            raise error(f"{name} has {problem[1]}, so it is not read as numbers")
        return value
    data = _unmasked(name, value, error)
    array = np.asarray(data)
    problem = _not_numbers(array)
    if problem is not None:
        position, why = problem
        raise error(
            f"{name} is {array.reshape(-1)[position]!r}{_where(array.shape, position)}: {why}, so "
            "it is not read as numbers"
        )
    return data


def _real_entries(
    data: Mapping[str, Any], names: Iterable[str], label: str = "data"
) -> dict[str, Any]:
    """``data`` with each entry ``names`` picks read by :func:`_real_numbers`, named
    ``label['name']``, and the others as they are: an entry no argument names is not read."""
    read = dict(data)
    for name in dict.fromkeys(names):
        read[name] = _real_numbers(data[name], f"{label}[{name!r}]")
    return read


def _not_numbers(column: NDArray[Any] | jax.Array) -> tuple[int, str] | None:
    """The first position, flat, of a value in ``column`` that is not read as a number, and why;
    ``None`` where every value is. ``column`` is a NumPy array, or a JAX array, traced or not, whose
    dtype alone decides.

    Read as numbers: a column of a boolean, an integer or a floating dtype, and an object column
    of ``bool``, ``int``, ``float``, ``Decimal`` or ``Fraction`` values, or NumPy's scalars of
    these kinds. Not read: text, dates, times, durations, complex numbers and any other object,
    whether a dtype or an object column holds them; and a number finite in its own type that is an
    infinity in float64, a ``Decimal`` of ``1E+400``, an ``int`` of 400 digits or a ``longdouble``
    past 1.8e308. A nan or an infinity is read as it is, as a float column's is: whether it is
    refused is the reader's rule, and a panel refuses it. A dtype NumPy holds as neither numbers
    nor text --- a structured one, or ``ml_dtypes``' ``bfloat16`` --- is left to the cast, which
    fails on the first and reads the second as the number it is.
    """
    if not column.size:
        return None
    kind = column.dtype.kind
    if kind in _NOT_REAL:
        return (
            0,
            f"dtype {column.dtype}, which holds {_NOT_REAL[kind]}, not real numbers",
        )
    if kind == "f" and column.dtype.itemsize > 8:  # only a float wider than float64 can pass it
        with np.errstate(over="ignore"):
            past = np.flatnonzero(np.isfinite(column) & np.isinf(column.astype(np.float64)))
        return (int(past[0]), _PAST_FLOAT64) if past.size else None
    if kind != "O":
        return None
    for position, value in enumerate(column.reshape(-1).tolist()):
        if not isinstance(value, _REAL) or isinstance(value, np.timedelta64):
            return position, f"type {type(value).__name__}, not a real number"
        if _past_float64(value):
            return position, _PAST_FLOAT64
    return None


def _past_float64(value: Any) -> bool:
    """Whether a real number that is finite in its own type is an infinity in float64."""
    if isinstance(value, Decimal):
        return value.is_finite() and math.isinf(float(value))
    if isinstance(value, float | np.floating):
        return bool(np.isfinite(value)) and math.isinf(float(value))
    try:  # an int, a Fraction, a bool or a NumPy integer, none of which has a nan or an infinity
        float(value)
    except OverflowError:  # an int or a Fraction past the range, where float() raises
        return True
    return False


def _at(shape: tuple[int, ...], position: int) -> str:
    """Where a flat position lies: a row of a column, an index of an array of more dimensions."""
    if len(shape) == 1:
        return f"row {position}"
    return f"index {tuple(int(i) for i in np.unravel_index(position, shape))}"


def _where(shape: tuple[int, ...], position: int) -> str:
    """`` at`` where a flat position lies, as :func:`_at` says; nothing for a single value."""
    return f" at {_at(shape, position)}" if shape else ""


def _by_name(pairs: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for label, column in pairs:
        name = str(label)
        if name in columns:
            raise ValueError(
                f"two columns go by the name {name!r}; chc reads a column by its name as text, so "
                "one would replace the other"
            )
        columns[name] = column
    return columns


def _refuse_shared_names(roles: Mapping[str, Iterable[str]]) -> None:
    """Refuse a column named in two of ``roles``, or twice in one.

    Each role reads the column its name picks, so a column named in two is read in both: a
    treatment that is also a covariate is partialled out of itself, and its effect reads as none.
    """
    seen: dict[str, str] = {}
    for role, names in roles.items():
        for name in names:
            if name not in seen:
                seen[name] = role
            elif seen[name] == role:
                raise ValueError(f"the column {name!r} is read twice as {role}; name it once")
            else:
                raise ValueError(
                    f"the column {name!r} is read as {seen[name]} and as {role}; each role needs "
                    "a column of its own"
                )
